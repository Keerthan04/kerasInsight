"use client";

import React, { useState } from "react";
import { useRouter } from "next/navigation";
import {
  Box,
  Heading,
  Text,
  Button,
  VStack,
  HStack,
  SimpleGrid,
  Modal,
  ModalOverlay,
  ModalContent,
  ModalHeader,
  ModalFooter,
  ModalBody,
  ModalCloseButton,
  Input,
  Textarea,
  useDisclosure,
  useToast,
  AlertDialog,
  AlertDialogBody,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogContent,
  AlertDialogOverlay,
} from "@chakra-ui/react";
import { Plus, Book, Trash2 } from "lucide-react";
import ChatWindow from "@/components/ChatWindow";

// Simulated data for existing knowledge bases
const initialKnowledgeBases = [
  { id: 1, name: "Keras Core API", endpointCount: 5 },
  { id: 2, name: "Keras Layers", endpointCount: 10 },
  { id: 3, name: "Keras Models", endpointCount: 3 },
];

export default function KnowledgeBaseDashboard() {
  const [knowledgeBases, setKnowledgeBases] = useState(initialKnowledgeBases);
  const { isOpen, onOpen, onClose } = useDisclosure();
  const [newKbName, setNewKbName] = useState("");
  const [newKbEndpoints, setNewKbEndpoints] = useState("");
  const [selectedKb, setSelectedKb] = useState(null);
  const [deleteKb, setDeleteKb] = useState(null);
  const [isDeleteAlertOpen, setIsDeleteAlertOpen] = useState(false);
  const toast = useToast();
  const router = useRouter();

  const handleCreateKnowledgeBase = () => {
    if (!newKbName.trim() || !newKbEndpoints.trim()) {
      toast({
        title: "Error",
        description: "Please provide both a name and endpoints.",
        status: "error",
        duration: 3000,
        isClosable: true,
      });
      return;
    }

    const newKb = {
      id: knowledgeBases.length + 1,
      name: newKbName,
      endpointCount: newKbEndpoints.split("\n").length,
    };

    setKnowledgeBases([...knowledgeBases, newKb]);
    setNewKbName("");
    setNewKbEndpoints("");
    onClose();

    toast({
      title: "Success",
      description: "New knowledge base created successfully.",
      status: "success",
      duration: 3000,
      isClosable: true,
    });
  };

  const handleOpenChat = (kb) => {
    setSelectedKb(kb);
  };

  const handleCloseChat = () => {
    setSelectedKb(null);
  };

  const handleOpenQueryPage = (kb) => {
    router.push(`/query/${kb.id}`);
  };

  const handleDeleteClick = (kb) => {
    setDeleteKb(kb);
    setIsDeleteAlertOpen(true);
  };

  const handleDeleteConfirm = () => {
    setKnowledgeBases(knowledgeBases.filter((kb) => kb.id !== deleteKb.id));
    setIsDeleteAlertOpen(false);
    setDeleteKb(null);
    toast({
      title: "Deleted",
      description: `${deleteKb.name} has been deleted.`,
      status: "info",
      duration: 3000,
      isClosable: true,
    });
  };

  const handleDeleteCancel = () => {
    setIsDeleteAlertOpen(false);
    setDeleteKb(null);
  };

  return (
    <Box maxWidth="container.xl" margin="auto" py={10}>
      <VStack spacing={8} align="stretch">
        <HStack justify="space-between">
          <Heading as="h1" size="xl" color="red.500">
            Knowledge Bases
          </Heading>
          <Button leftIcon={<Plus />} colorScheme="red" onClick={onOpen}>
            Create New
          </Button>
        </HStack>

        <SimpleGrid columns={{ base: 1, md: 2, lg: 3 }} spacing={6}>
          {knowledgeBases.map((kb) => (
            <Box key={kb.id} borderWidth={1} borderRadius="lg" p={6}>
              <VStack align="stretch" spacing={3}>
                <HStack>
                  <Book size={24} color="red" />
                  <Heading size="md">{kb.name}</Heading>
                </HStack>
                <Text>{kb.endpointCount} endpoints</Text>
                <Button
                  colorScheme="red"
                  variant="outline"
                  onClick={() => handleOpenChat(kb)}
                >
                  Quick Query
                </Button>
                <Button
                  colorScheme="blue"
                  variant="outline"
                  onClick={() => handleOpenQueryPage(kb)}
                >
                  Full Query Page
                </Button>
                <Button
                  leftIcon={<Trash2 />}
                  colorScheme="red"
                  variant="ghost"
                  onClick={() => handleDeleteClick(kb)}
                >
                  Delete
                </Button>
              </VStack>
            </Box>
          ))}
        </SimpleGrid>
      </VStack>

      <Modal isOpen={isOpen} onClose={onClose}>
        <ModalOverlay />
        <ModalContent>
          <ModalHeader>Create New Knowledge Base</ModalHeader>
          <ModalCloseButton />
          <ModalBody>
            <VStack spacing={4}>
              <Input
                placeholder="Knowledge Base Name"
                value={newKbName}
                onChange={(e) => setNewKbName(e.target.value)}
              />
              <Textarea
                placeholder="Enter API endpoints (one per line)"
                value={newKbEndpoints}
                onChange={(e) => setNewKbEndpoints(e.target.value)}
                rows={6}
              />
            </VStack>
          </ModalBody>
          <ModalFooter>
            <Button
              colorScheme="red"
              mr={3}
              onClick={handleCreateKnowledgeBase}
            >
              Create
            </Button>
            <Button variant="ghost" onClick={onClose}>
              Cancel
            </Button>
          </ModalFooter>
        </ModalContent>
      </Modal>

      <AlertDialog
        isOpen={isDeleteAlertOpen}
        leastDestructiveRef={null}
        onClose={handleDeleteCancel}
      >
        <AlertDialogOverlay>
          <AlertDialogContent>
            <AlertDialogHeader fontSize="lg" fontWeight="bold">
              Delete Knowledge Base
            </AlertDialogHeader>

            <AlertDialogBody>
              Are you sure you want to delete {deleteKb?.name}? This action
              cannot be undone.
            </AlertDialogBody>

            <AlertDialogFooter>
              <Button onClick={handleDeleteCancel}>Cancel</Button>
              <Button colorScheme="red" onClick={handleDeleteConfirm} ml={3}>
                Delete
              </Button>
            </AlertDialogFooter>
          </AlertDialogContent>
        </AlertDialogOverlay>
      </AlertDialog>

      {selectedKb && (
        <ChatWindow
          knowledgeBase={selectedKb}
          isOpen={!!selectedKb}
          onClose={handleCloseChat}
        />
      )}
    </Box>
  );
}
